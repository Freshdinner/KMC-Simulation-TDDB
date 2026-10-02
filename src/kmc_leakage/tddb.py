"""TDDB 瞬态驱动：准静态双层循环，含热反馈与击穿判据。

规格见 PROJECT_CONTEXT_TDDB.md §5、§6，决策记录见 ADR-032–042。公式编号 T1–T14 是 TDDB 的
编号（PROJECT_CONTEXT_TDDB.md 第 3 节），与 docs/rate_equations.md 的 R1–R8 不对应。

电子输运事件的时间尺度（~10⁻¹³…10⁻⁶ s）比缺陷生成（秒以上）短十几个数量级，所以输运相对
生成瞬间达到稳态（§5）。外层是缺陷生成的 Gillespie 循环，每步：

1. **内层稳态**：按当前缺陷集合与输运温度重建 `KMCSimulation`（静电耦合用
   `IncrementalCoupling` 逐个追加的那份），从上一步的占据热启动，求簇级平均场不动点
   （`mean_field.solve_mean_field`），得到电流、分解与各缺陷占据 p。近邻表、强耦合簇与截断
   半径都随缺陷集合从头重算，与对同一缺陷集合新建的 `KMCSimulation` 逐项相同（ADR-033）。
2. **功率与温度**（式 T13/T14，`thermal.py`）：在内层稳态上逐事件累加放热 ΔE·R，介质内的
   功率解稳态热方程得到温度场（两电极恒温 T_amb）。配置了 k_TH 才开热反馈：温度场在生成
   单元中心与陷阱项积分节点上的值进生成率，缺陷处按放热加权的平均温度作为下一步内层输运的全局标量
   温度 T(t)（§6.6）。输运与热不迭代自洽，每步只走一遍"输运 → 功率 → 温度"（§8.4），
   所以第 k 条记录的电流是在第 k−1 条记录的功率给出的温度下求的；温度随生成一步步爬升，
   这一步的滞后可以忽略。没有 k_TH 时温度固定为环境值（Phase 3 的等温口径）。
3. **生成率表**（§6.1 模型 C）：各缺陷净电荷 q·(Z − p) 与外加偏压一起解一次泊松方程。式 T4 的
   R_1e 按电子来源分成两类事件（ADR-042）：
   - **电极项**（两侧电极 R2 + R4）：在每个生成单元中心对一个虚拟前驱求值，乘断键概率 P_bb
     （式 T2），单元生成率 G_cell = N_T·V_cell·R_1e,电极·P_bb（式 T1）。单元中心离已有缺陷不到
     min_separation 的单元放不下新空位，生成率记 0；
   - **陷阱项**（近旁已占据陷阱经陷阱间通道 p·R7 送来的电子，ADR-039）：几何因子沿距离每
     r_D/2 ≈ 0.083 nm 衰减 e 倍，远小于生成单元，所以不放在单元上，而是逐个源陷阱在它周围的
     可放置区域里对 N_T·p·R7·P_bb 做体积分（`TrapCaptureGeneration`），每个源陷阱一条事件。
4. **击穿判据**（§6.3）：间距 < d_c 的缺陷连边，某个连通分量同时触到两个电极（离电极界面
   < d_c）即击穿，这一刻就是 t_BD；电流密度达到限流值是第二停止条件。论文的"25 个空位、
   间距 ≤ 6 Å"团簇只作标志记录，不作判据。
5. **抽事件**：Δt = −ln(r)/ΣG，在单元事件与源陷阱事件里按速率选一条。选中单元时在单元内连续
   随机落点，选中源陷阱时按它周围的被积函数抽落点；与已有缺陷的距离小于 min_separation 就重抽
   （§6.1、§6.8）。
6. **出生**：新空位与初始缺陷同一物种（能级按缺陷预设的分布抽样，S、m_di 取预设），出生时
   被前驱俘获的那个电子占据，电荷态为 0（中性）；电子发射后空位才带 +1（施主约定，代码里
   `charge_state` = +1 是电子离开后的电荷，§6.7、ADR-035）。所以它出生这一刻不改变其余缺陷
   处的电势与电场。静电耦合增量扩一行一列（§6.4），回到 1。

未击穿时按生成空位数上限或应力时间上限停止。

**电荷记账**：新空位出生时带的那个电子来自前驱俘获，不经过电极，不计入输运电流（§6.7）。
平均场没有时间推进，每条记录的电流就是该缺陷集合下的稳态电流。
"""

import csv
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
from scipy import constants
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .bands import (
    anode_fermi_level,
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
    minimum_image_delta,
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
    bond_break_generation_rate,
    bond_break_probability,
    effective_activation_energy,
    single_electron_capture_rate,
    trap_precursor_capture_rate,
)
from .thermal import heat_release, power_map, solve_temperature, transport_temperature

# 前驱俘获 R4 的声子项数：E_rel1 = 1.0 eV（E_rel1/ℏω₀ = 25）时 80 项最大相对误差 1e-4、
# 100 项到机器精度（ADR-030）
_PRECURSOR_N_PHONON = 100
# 论文的致密团簇标志"25 个空位、间距 ≤ 6 Å"（§6.3）：只记录最大团簇的大小与首次达到的时刻，
# 不作击穿判据（那是针对 ~2.7 nm 膜定的）
_DENSE_CLUSTER_DISTANCE = 0.6 * constants.nano
_DENSE_CLUSTER_SIZE = 25
# 生成单元分块求 R_1e：每块 (块长, 声子数, 路径点) 的数组约 0.4 MB，留在缓存里。16×16×70 单元上
# 实测单进程 256 块 4.3 s、8 块 3.6 s；16 个进程并发时 256 块 36 s（内存带宽占满）、8 块 9–14 s
_CELL_CHUNK = 8
# 选中单元里落点的尝试次数上限，全部离已有缺陷太近就把单元记为已满
_MAX_PLACEMENT_ATTEMPTS = 1000
# 陷阱项的源陷阱球坐标积分（`TrapCaptureGeneration`，ADR-042）。极角 u = cosθ 方向在 MA 因子与 P_bb
# 饱和的折点、可放置区域边界处切开后按对数线性精确积分，径向与方位角 φ 在相邻节点之间按对数线性插值积分：
# 几何因子 e^{−α·r} 与外场下 MA 因子的指数变化都没有离散误差，误差随三个步长二阶下降。取值由 ADR-042
# 的收敛性测试定：孤立源陷阱对解析参考的误差 ≤ 5×10⁻³，致密空位层上逐项加密一倍 ΣG 变化 ≤ 2%。
_TRAP_RADIAL_STEP = 0.04 * constants.nano
_TRAP_POLAR_BINS = 24
_TRAP_AZIMUTH_BINS = 24
# 径向截断：截掉部分的上界（Θ、P_bb 都取 1）不超过已积出部分的这个比例
_TRAP_TAIL_TOLERANCE = 1.0e-3
_TRAP_RADIAL_CHUNK = 8  # 每轮向外推进的径向区间数，之后逐个源陷阱检查截断判据
_TRAP_SOURCE_CHUNK = 32  # 每块源陷阱数：块内 (源, 极角, 方位角, 径向) 的节点约 1.7×10⁵ 个


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
class GenerationCells:
    """生成单元表（§6.1 模型 C）：铺满 [0,Lx)×[0,Ly)×[standoff, Lz−standoff] 的长方体网格。

    z 方向只覆盖可采样窗口：两端 standoff（min_separation）内按数值稳定性要求不放缺陷，
    与 `generate_defects` 的口径相同。单元按 (ix, iy, iz) 的 C 序排列。
    """

    lower: np.ndarray  # (M, 3) 各单元的下角坐标 [m]
    size: np.ndarray  # (3,) 单元边长 [m]
    layer: np.ndarray  # (M,) 单元所在的 z 层号（同一层的单元共用电极→单元的 WKB 路径）
    z_centers: np.ndarray  # (nz,) 各 z 层中心 [m]

    def __len__(self) -> int:
        return int(self.lower.shape[0])

    @property
    def centers(self) -> np.ndarray:
        """各单元中心 (M, 3) [m]：虚拟前驱的求值点。"""
        return self.lower + 0.5 * self.size

    @property
    def volume(self) -> float:
        """单元体积 V_cell [m^3]。"""
        return float(np.prod(self.size))


def generation_cells(device: DeviceConfig, grid: Tuple[int, int, int]) -> GenerationCells:
    """按 `grid` = (nx, ny, nz) 把可采样窗口切成生成单元。"""
    standoff = _global_standoff(device)
    thickness = device.Lz - 2.0 * standoff
    if thickness <= 0.0:
        raise ValueError("min_separation 约束下 z 方向没有可放置新空位的区间")
    gx, gy, gz = grid
    size = np.array([device.Lx / gx, device.Ly / gy, thickness / gz])
    ix, iy, iz = np.meshgrid(np.arange(gx), np.arange(gy), np.arange(gz), indexing="ij")
    lower = np.stack(
        [ix.ravel() * size[0], iy.ravel() * size[1], standoff + iz.ravel() * size[2]], axis=1
    )
    return GenerationCells(
        lower=lower,
        size=size,
        layer=iz.ravel(),
        z_centers=standoff + (np.arange(gz) + 0.5) * size[2],
    )


def admissible_cells(cells: GenerationCells, positions: np.ndarray, device: DeviceConfig) -> np.ndarray:
    """单元中心离全部已有缺陷都不小于 min_separation 的单元（xy 取最小镜像）。

    新空位不能落在已有缺陷的 min_separation 之内（§6.1），那里的生成速率就是 0。
    """
    standoff = _global_standoff(device)
    if positions.shape[0] == 0 or standoff <= 0.0:
        return np.ones(len(cells), dtype=bool)
    # xy 周期：坐标先折回 [0, L)；z 不周期：盒长取 4·Lz，坐标都在 [0, Lz] 内，不会回绕
    box = np.array([device.Lx, device.Ly, 4.0 * device.Lz])
    tree = cKDTree(np.mod(positions, box), boxsize=box)
    distance, _ = tree.query(np.mod(cells.centers, box))
    return distance >= standoff


def _log_linear_integral(start: np.ndarray, end: np.ndarray, width: np.ndarray) -> np.ndarray:
    """端点值为 `start`、`end`、宽度 `width` 的区间上，按对数线性插值（区间内取指数函数）的精确积分。

    被积函数本身是指数函数时没有离散误差。一端为 0（落在不可放置的区域里）时退回梯形。
    """
    start, end, width = np.broadcast_arrays(
        np.asarray(start, dtype=float), np.asarray(end, dtype=float), np.asarray(width, dtype=float)
    )
    result = 0.5 * (start + end) * width
    positive = (start > 0.0) & (end > 0.0)
    a, b, w = start[positive], end[positive], width[positive]
    d = np.log(b) - np.log(a)
    small = np.abs(d) < 1e-6
    # (b − a)/d 就是 a·(e^d − 1)/d，写成差的形式不会在 e^d 上溢出
    result[positive] = w * np.where(small, a * (1.0 + 0.5 * d), (b - a) / np.where(small, 1.0, d))
    return result


def _kinked_log_linear_integral(
    log_a: np.ndarray,
    log_b: np.ndarray,
    kinks_a: Tuple[np.ndarray, ...],
    kinks_b: Tuple[np.ndarray, ...],
    start: np.ndarray,
    stop: np.ndarray,
    width: float,
) -> np.ndarray:
    """width·∫_start^stop exp(L(t)) dt，t ∈ [0, 1] 是区间内的分数位置，
    L(t) = (1−t)·log_a + t·log_b − Σ_m max(0, (1−t)·h_m,a + t·h_m,b)：光滑部分的对数与每个折点指数 h_m
    （MA 因子与 P_bb 的饱和）都线性插值。

    每个 h_m 在 [start, stop] 内变号处切开，每段上 L 线性，按对数线性精确积分。没有折点落在区间内的
    行不切，一段积完。
    """
    kinks = []
    for h_a, h_b in zip(kinks_a, kinks_b):
        crosses = (h_a > 0.0) != (h_b > 0.0)
        kinks.append(np.where(crosses, h_a / np.where(crosses, h_a - h_b, 1.0), stop))
    split = np.zeros(start.shape, dtype=bool)
    for kink in kinks:
        split |= (kink > start) & (kink < stop)

    def value(t: np.ndarray, rows) -> np.ndarray:
        log = log_a[rows] + t * (log_b[rows] - log_a[rows])
        for h_a, h_b in zip(kinks_a, kinks_b):
            log = log - np.maximum(0.0, h_a[rows] + t * (h_b[rows] - h_a[rows]))
        return np.exp(log)

    whole = ~split
    result = np.zeros(start.shape)
    result[whole] = _log_linear_integral(
        value(start[whole], whole), value(stop[whole], whole), width * (stop[whole] - start[whole])
    )
    if split.any():
        cuts = np.sort(np.stack([start[split], stop[split]] + [
            np.clip(kink[split], start[split], stop[split]) for kink in kinks
        ]), axis=0)
        ends = [value(t, split) for t in cuts]
        result[split] = sum(
            _log_linear_integral(ends[m], ends[m + 1], width * (cuts[m + 1] - cuts[m]))
            for m in range(len(cuts) - 1)
        )
    return result


def _log_linear_fraction(start: float, end: float, uniform: float) -> float:
    """[0, 1] 上按对数线性密度（两端值 `start`、`end`）抽样：均匀随机数 `uniform` 的逆变换。

    与 `_log_linear_integral` 同一个插值模型；一端为 0 时退回均匀分布。
    """
    if start > 0.0 and end > 0.0:
        slope = float(np.clip(np.log(end) - np.log(start), -700.0, 700.0))
        if abs(slope) > 1e-9:
            return float(np.log1p(uniform * np.expm1(slope)) / slope)
    return uniform


class TrapCaptureGeneration:
    """R_1e 陷阱项的生成事件：逐个源陷阱对它周围的生成率做体积分（式 T1/T4/T5，ADR-039、ADR-042）。

    源陷阱 i 的事件速率

        G_i = N_T·p_i·∫_A R7(i → x)·P_bb(x) d³x,
        R7(i → x) = ν_i·exp[−|x − x_i|·(1/r_D,i + 1/r_D,前驱)]·Θ(E_前驱(x) − E_i),

    就是 ADR-039 的单元中心求和 Σ_单元 N_T·V_cell·P_bb·Σ_i p_i·R7 在单元无限细时的极限，被积函数逐点
    的取法不变：前驱能级 E_前驱(x) = B − e·φ(x) − E_t1 与局域场 |E(x)| 取当前占据下含空间电荷的局域值，
    E_i 取源陷阱的平均场能级（ADR-038），温度取 x 处的局域温度（§6.6），P_bb 的 k 取静态 κ（式 T2）。
    A 是可放置区域：z ∈ [standoff, Lz − standoff]，且离全部已有缺陷 ≥ min_separation（xy 最小镜像）。

    积分在以源陷阱为中心的球坐标 (r, u = cosθ, φ) 上做，r 从 min_separation 起每 `_TRAP_RADIAL_STEP`
    取一个节点。几何因子的 e 折长度只有 r_D/2，外场下 MA 因子沿 r·u 同样按指数变化（参考工况每 nm 约
    e^{14.6}，可以比几何衰减还快，深能级源陷阱的被积函数峰值因此落在下游、离开 min_separation），而
    MA 因子与 P_bb 的饱和都是折点，峰值常落在折点上。所以先在每个径向节点的球面上沿极角积分
    （`_shells`：折点与可放置区域的边界都按线性插值求交点切开，每段对数线性精确积分），再沿径向在
    相邻节点之间按对数线性插值积分，方位角同样在相邻节点之间按对数线性插值（周期）。径向逐轮向外推进，截掉部分的上界
    4π·ν_i·∫_r^∞ r'²·e^{−α_i·r'} dr'（Θ、P_bb 都取 1，α_i = 1/r_D,i + 1/r_D,前驱）不超过已积出部分的
    `_TRAP_TAIL_TOLERANCE` 倍时停下；半径不超过横向盒长的一半，球不会与自己的周期镜像重叠。p_i 是整体
    前置因子、不进积分，所以截断位置与 p_i、N_T 都无关。
    """

    def __init__(
        self,
        inner: KMCSimulation,
        occupancy: np.ndarray,
        tddb: TDDBConfig,
        phi: np.ndarray,
        temperature_field: Optional[np.ndarray],
    ) -> None:
        device, solver, defects = inner.device, inner.solver, inner.defects
        material = device.stack.single_material
        self._device = device
        self._solver = solver
        # 被积函数要读的网格量叠成一个多通道数组，每个节点一次插值读回：φ、E 的三个分量、（热反馈时）温度
        self._grid = np.stack(
            (phi,) + solver.electric_field(phi) + (() if temperature_field is None else (temperature_field,)),
            axis=-1,
        )
        self._temperature = inner.sim.temperature  # 没有温度场时的工作点温度
        self._tddb = tddb
        self._kappa = material.kappa
        self._precursor_r_D = localization_radius(tddb.precursor_ionization_energy, material.m_di)
        self._positions = defects.positions
        self._r_D = defects.r_D
        self._attempt_frequency = defects.attempt_frequency
        self._standoff = _global_standoff(device)
        # xy 周期：坐标先折回 [0, L)；z 不周期：盒长取 4·Lz，可放置区域的点都在 [0, Lz] 内，不会回绕
        self._box = np.array([device.Lx, device.Ly, 4.0 * device.Lz])
        self._tree = cKDTree(np.mod(defects.positions, self._box), boxsize=self._box)

        self._u_edges = np.linspace(-1.0, 1.0, _TRAP_POLAR_BINS + 1)
        azimuth = np.arange(_TRAP_AZIMUTH_BINS) * (2.0 * np.pi / _TRAP_AZIMUTH_BINS)  # 方位角节点（周期）
        sin_theta = np.sqrt(1.0 - self._u_edges ** 2)[:, None]
        self._directions = np.stack(
            np.broadcast_arrays(
                sin_theta * np.cos(azimuth)[None, :], sin_theta * np.sin(azimuth)[None, :], self._u_edges[:, None]
            ),
            axis=-1,
        )  # (极角棱, 方位角, 3)

        p = np.asarray(occupancy, dtype=float)
        n = len(defects)
        self.n_intervals = np.zeros(n, dtype=int)  # 各源陷阱积到的径向区间数
        self._sampling: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}  # 抽样用的分格积分
        integral = np.zeros(n)
        sources = np.flatnonzero(p > 0.0) if tddb.precursor_concentration > 0.0 else np.zeros(0, dtype=int)
        if sources.size:
            self._trap_energy = inner._trap_state(p)[0]
            integral[sources], self.n_intervals[sources] = self._integrate(sources)
        self.rates = tddb.precursor_concentration * p * integral  # (N,) 各源陷阱的事件速率 G_i [1/s]

    def _integrate(self, sources: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """各源陷阱的 ∫_A R7·P_bb d³x [m³/s]（不含 p_i）与积到的径向区间数，径向按截断判据逐轮推进。"""
        alpha = 1.0 / self._r_D[sources] + 1.0 / self._precursor_r_D
        nu = self._attempt_frequency[sources]
        half_box = 0.5 * min(self._device.Lx, self._device.Ly)
        integral = np.zeros(sources.size)
        n_intervals = np.zeros(sources.size, dtype=int)
        active = np.arange(sources.size)
        start = 0
        while active.size:
            radii = self._standoff + _TRAP_RADIAL_STEP * np.arange(start, start + _TRAP_RADIAL_CHUNK + 1)
            for block in np.array_split(active, int(np.ceil(active.size / _TRAP_SOURCE_CHUNK))):
                shells, _ = self._shells(sources[block], radii)
                integral[block] += self._bin_weights(shells).sum(axis=(1, 2, 3))
            start += _TRAP_RADIAL_CHUNK
            r, a = radii[-1], alpha[active]
            tail = 4.0 * np.pi * nu[active] * np.exp(-a * r) * (r ** 2 / a + 2.0 * r / a ** 2 + 2.0 / a ** 3)
            done = (tail <= _TRAP_TAIL_TOLERANCE * integral[active]) | (
                r + _TRAP_RADIAL_CHUNK * _TRAP_RADIAL_STEP > half_box
            )
            n_intervals[active[done]] = start
            active = active[~done]
        return integral, n_intervals

    def _shells(self, sources: np.ndarray, radii: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """每个径向节点上逐个极角区间的积分 J = ∫_A r²·R7·P_bb du（不含 p_i），形状 (源, 极角区间, 方位角,
        径向节点)；以及节点上的被积函数值 (源, 极角棱, 方位角, 径向节点)。

        被积函数 r²·R7·P_bb = exp[S − max(0, h) − max(0, g)] 有两个折点：MA 因子 Θ = exp(−max(0, h))，
        h = ΔE/k_BT；P_bb = exp(−max(0, g))，g = (E_A − p₀·(2+k)/3·E)/k_BT（有效势垒降到 0 以下饱和）。
        被积函数的峰值常常就落在折点上（深能级源陷阱的 Θ 刚到 1 处、带电缺陷近旁 P_bb 刚饱和处）。
        极角区间内 S、h、g 都按两条棱射线上的节点值线性插值，h、g 变号处切开。区间一条棱可放置、一条不可放置时，按边界函数（到可采样窗口平面的
        距离、到挡住那条棱的缺陷的距离减 min_separation）的线性插值求交点，只积可放置的一段。每一段上
        ln 被积函数是线性的，按对数线性精确积分。这样 J 随 r 连续变化，径向再按对数线性插值积分。
        """
        points = (
            self._positions[sources][:, None, None, None, :]
            + radii[None, None, None, :, None] * self._directions[None, :, :, None, :]
        )
        shape = points.shape[:-1]
        flat = points.reshape(-1, 3)
        source = np.broadcast_to(sources[:, None, None, None], shape).ravel()
        admissible, blocker = self._admissible(flat, source)
        admissible, blocker = admissible.reshape(shape), blocker.reshape(shape)
        # 被积函数在可放置节点与它们沿极角相邻的节点上求值（半可放置区间要用到两条棱的值）
        need = admissible.copy()
        need[:, 1:] |= admissible[:, :-1]
        need[:, :-1] |= admissible[:, 1:]
        rows = need.ravel()
        distance = np.broadcast_to(radii, shape).ravel()
        values, uphill, barrier = (np.zeros(flat.shape[0]) for _ in range(3))
        values[rows], uphill[rows], barrier[rows] = self._integrand(source[rows], flat[rows], distance[rows])
        values, uphill, barrier = values.reshape(shape), uphill.reshape(shape), barrier.reshape(shape)
        # 光滑部分取对数：ln(r²·R7·P_bb) + max(0, h) + max(0, g)；下溢的值夹在最小正数上，exp 回去仍是 0
        smooth = np.log(np.maximum(values, np.finfo(float).tiny)) + np.maximum(uphill, 0.0) + np.maximum(barrier, 0.0)

        lower, upper = (slice(None), slice(None, -1)), (slice(None), slice(1, None))
        ok_a, ok_b = admissible[lower], admissible[upper]
        # 积分范围（区间内的分数位置）：两条棱都可放置时 [0, 1]，都不可放置时为空
        start, stop = np.where(ok_a & ok_b, 0.0, 1.0), np.ones(ok_a.shape)
        mixed = ok_a ^ ok_b
        if mixed.any():
            s, b, a, k = np.nonzero(mixed)
            from_a = ok_a[mixed]  # True：可放置的是下棱 b
            edge_ok = np.where(from_a, b, b + 1)
            edge_out = np.where(from_a, b + 1, b)
            fraction = self._crossing(
                points[s, edge_ok, a, k], points[s, edge_out, a, k], blocker[s, edge_out, a, k]
            )
            start[mixed] = np.where(from_a, 0.0, 1.0 - fraction)
            stop[mixed] = np.where(from_a, fraction, 1.0)
        shells = np.zeros(start.shape)
        open_ = stop > start
        shells[open_] = _kinked_log_linear_integral(
            smooth[lower][open_],
            smooth[upper][open_],
            (uphill[lower][open_], barrier[lower][open_]),
            (uphill[upper][open_], barrier[upper][open_]),
            start[open_],
            stop[open_],
            2.0 / _TRAP_POLAR_BINS,
        )
        return shells, values

    def _crossing(self, inside: np.ndarray, outside: np.ndarray, blocker: np.ndarray) -> np.ndarray:
        """从可放置的 `inside` 到不可放置的 `outside`，可放置区域边界所在的分数位置 ∈ [0, 1]。

        边界函数 g（`outside` 在可采样窗口外时取到窗口平面的有向距离，落在缺陷 `blocker` 的
        min_separation 球内时取到它的距离减 min_separation）在 `inside` 上 ≥ 0、`outside` 上 < 0，按两端
        线性插值求零点；两种都有时取离 `inside` 近的那个。
        """
        device, standoff = self._device, self._standoff
        fraction = np.ones(inside.shape[0])
        for g_in, g_out in (
            (inside[:, 2] - standoff, outside[:, 2] - standoff),
            (device.Lz - standoff - inside[:, 2], device.Lz - standoff - outside[:, 2]),
        ):
            cut = g_out < 0.0
            fraction[cut] = np.minimum(fraction[cut], g_in[cut] / (g_in[cut] - g_out[cut]))
        hit = blocker >= 0
        if hit.any():
            center = self._positions[blocker[hit]]
            g_in = np.linalg.norm(minimum_image_delta(inside[hit] - center, device.Lx, device.Ly), axis=-1) - standoff
            g_out = np.linalg.norm(minimum_image_delta(outside[hit] - center, device.Lx, device.Ly), axis=-1) - standoff
            fraction[hit] = np.minimum(fraction[hit], g_in / (g_in - g_out))
        return np.clip(fraction, 0.0, 1.0)

    def _admissible(self, points: np.ndarray, source: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """点是否在可放置区域 A 里，以及挡住它的缺陷编号（离它 < min_separation 的最近缺陷，没有为 −1）。

        A：可采样窗口内，且离源陷阱以外的全部缺陷 ≥ min_separation。到源陷阱自己的距离就是节点
        半径，按构造 ≥ min_separation，这里按编号排除，免得舍入误差把 r = min_separation 那一圈判成
        太近。窗口外的点不查近邻，挡住它的是窗口平面。
        """
        z = points[:, 2]
        keep = (z >= self._standoff) & (z <= self._device.Lz - self._standoff)
        blocker = np.full(points.shape[0], -1)
        candidates = np.flatnonzero(keep)
        k = min(2, self._tree.n)
        distance, index = self._tree.query(
            np.mod(points[candidates], self._box), k=k, distance_upper_bound=self._standoff
        )
        index = index.reshape(candidates.size, k)
        distance = np.where(index == source[candidates, None], np.inf, distance.reshape(candidates.size, k))
        nearest = np.argmin(distance, axis=1)
        blocked = distance[np.arange(candidates.size), nearest] < self._standoff
        keep[candidates] = ~blocked
        blocker[candidates[blocked]] = index[blocked, nearest[blocked]]
        return keep, blocker

    def _integrand(
        self, source: np.ndarray, points: np.ndarray, distance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(r²·R7(i → x)·P_bb(x), h, g)：前驱能级、局域场与温度都取 x 处的局域值（取法同单元中心的
        电极项）。h = (E_前驱(x) − E_i)/k_BT，R7 里的 MA 因子是 exp(−max(0, h))；g = (E_A − p₀·(2+k)/3·E)/k_BT，
        P_bb = exp(−max(0, g))。"""
        tddb = self._tddb
        local = self._solver.interpolate(self._grid, points)
        precursor_energy = (
            self._device.band_reference_profile(points[:, 2])
            - constants.elementary_charge * local[:, 0]
            - tddb.precursor_ionization_energy
        )
        temperature = local[:, 4] if local.shape[1] > 4 else self._temperature
        field = np.sqrt(local[:, 1] ** 2 + local[:, 2] ** 2 + local[:, 3] ** 2)
        trap_energy = self._trap_energy[source]
        hop = trap_precursor_capture_rate(
            self._attempt_frequency[source],
            distance,
            self._r_D[source],
            self._precursor_r_D,
            trap_energy,
            precursor_energy,
            1.0,
            temperature,
        )
        p_bb = bond_break_probability(tddb.activation_energy, tddb.dipole_moment, self._kappa, field, temperature)
        kT = constants.Boltzmann * temperature
        barrier = effective_activation_energy(tddb.activation_energy, tddb.dipole_moment, self._kappa, field) / kT
        return distance ** 2 * hop * p_bb, (precursor_energy - trap_energy) / kT, barrier

    @staticmethod
    def _azimuth_integrals(shells: np.ndarray) -> np.ndarray:
        """每个径向节点上逐个 (极角区间, 方位角区间) 的积分，形状同 `shells`：方位角区间 j 在节点 j 与
        j+1（周期）的极角区间积分之间按对数线性插值。带电缺陷近旁 P_bb ≈ exp(p₀·(2+k)/3·|E|/k_BT) 把局域场
        的小起伏放大成指数变化，对数线性插值对指数变化没有离散误差。"""
        return _log_linear_integral(shells, np.roll(shells, -1, axis=-2), 2.0 * np.pi / _TRAP_AZIMUTH_BINS)

    def _bin_weights(self, shells: np.ndarray) -> np.ndarray:
        """每个 (极角区间, 方位角区间, 径向区间) 的积分，形状 (源, 极角区间, 方位角区间, 径向区间)：
        方位角见 `_azimuth_integrals`，径向在相邻两个节点之间按对数线性插值。"""
        sectors = self._azimuth_integrals(shells)
        return _log_linear_integral(sectors[..., :-1], sectors[..., 1:], _TRAP_RADIAL_STEP)

    def sample(self, source: int, rng: np.random.Generator) -> Optional[np.ndarray]:
        """按源陷阱 `source` 周围的被积函数抽一个新空位坐标；`_MAX_PLACEMENT_ATTEMPTS` 次都放不下返回 None。

        先按积分权重选 (极角区间, 方位角区间, 径向区间)，区间内 r、φ、u 依次按两端的积分或节点值做对数
        线性密度抽样（r：方位角区间积分；φ：两个方位角节点上的极角区间积分；u：两条棱上的被积函数值），
        与求积时的插值同一个模型；落点再按可采样窗口与 min_separation 严格检查，不满足就重抽。同一个
        源陷阱的分格积分算一次、留着复用。
        """
        device = self._device
        if source not in self._sampling:
            radii = self._standoff + _TRAP_RADIAL_STEP * np.arange(self.n_intervals[source] + 1)
            shells, values = self._shells(np.array([source]), radii)
            sectors = self._azimuth_integrals(shells)[0]
            cumulative = np.cumsum(_log_linear_integral(sectors[..., :-1], sectors[..., 1:], _TRAP_RADIAL_STEP).ravel())
            self._sampling[source] = (radii, shells[0], sectors, values[0], cumulative)
        radii, shells, sectors, values, cumulative = self._sampling[source]
        shape = sectors.shape[:-1] + (sectors.shape[-1] - 1,)  # (极角区间, 方位角区间, 径向区间)
        if not cumulative[-1] > 0.0:
            return None
        for _ in range(_MAX_PLACEMENT_ATTEMPTS):
            flat = int(np.searchsorted(cumulative, rng.random() * cumulative[-1], side="right"))
            b, j, k = np.unravel_index(min(flat, cumulative.size - 1), shape)
            step = _log_linear_fraction(sectors[b, j, k], sectors[b, j, k + 1], rng.random())
            r = radii[k] + _TRAP_RADIAL_STEP * step
            node = k + int(round(step))
            turn = _log_linear_fraction(shells[b, j, node], shells[b, (j + 1) % shape[1], node], rng.random())
            azimuth = (j + turn) * (2.0 * np.pi / _TRAP_AZIMUTH_BINS)
            edge = (j + int(round(turn))) % shape[1]
            u = self._u_edges[b] + (2.0 / _TRAP_POLAR_BINS) * _log_linear_fraction(
                values[b, edge, node], values[b + 1, edge, node], rng.random()
            )
            sin_theta = np.sqrt(max(0.0, 1.0 - u * u))
            point = self._positions[source] + r * np.array(
                [sin_theta * np.cos(azimuth), sin_theta * np.sin(azimuth), u]
            )
            point[:2] = np.mod(point[:2], [device.Lx, device.Ly])
            if self._standoff <= point[2] <= device.Lz - self._standoff:
                delta = minimum_image_delta(self._positions - point[None, :], device.Lx, device.Ly)
                if np.min(np.linalg.norm(delta, axis=-1)) >= self._standoff:
                    return point
        return None


class GenerationTable(NamedTuple):
    """一个状态下的生成事件表（式 T1，§6.1 模型 C，ADR-042）：单元事件在前、源陷阱事件在后。"""

    cells: np.ndarray  # (M,) 各生成单元的电极项事件速率 G_cell [1/s]
    traps: TrapCaptureGeneration  # 各源陷阱的陷阱项事件，速率 `traps.rates` (N,) [1/s]


def generation_rates(
    inner: KMCSimulation,
    occupancy: np.ndarray,
    tddb: TDDBConfig,
    cells: GenerationCells,
    temperature_field: Optional[np.ndarray] = None,
) -> GenerationTable:
    """O 空位生成事件表（式 T1，§6.1 模型 C）：R_1e 的电极项按生成单元、陷阱项按源陷阱（ADR-042）。

    `inner` 提供当前缺陷集合、工作点与泊松求解器，`occupancy` 是各缺陷的平均场占据 p。
    每个缺陷的净电荷 q·(Z − p) 与外加偏压一起解一次泊松方程（与引擎用耦合矩阵线性叠加的
    结果相同），两类事件都在这份电势上求值：

    - **单元事件** G_cell = N_T·V_cell·R_1e,电极·P_bb，在单元中心对一个虚拟前驱求值，取法同
      `kmc.KMCSimulation._rates` 处理真实陷阱：前驱处导带底 E_C = B − e·φ 定能级 E = E_C − E_t1，
      局域场大小进 R4 的 c_MP 与 P_bb；电极→单元的 WKB 路径沿 z 直线，势垒取横向平均剖面的导带底；
      R_1e,电极 = Σ_两侧电极 (R2 + R4)（式 T4，`single_electron_capture_rate`，R4 取
      `_PRECURSOR_N_PHONON` 个声子项）；P_bb 的场加速项 p₀·(2+k)/3·E 里 k 取静态 κ（式 T2）。
      单元中心落在已有缺陷 min_separation 之内的单元记 0（`admissible_cells`）：那里放不下新空位，
      而带电缺陷近旁的库仑场经 (2+κ)/3 放大后会把 P_bb 压到饱和，把这个速率算给整个单元会让
      ΣG 随生成网格加密发散（ADR-033）。
    - **源陷阱事件**：近旁已占据陷阱经陷阱间通道 p·R7 送给前驱的电子（式 T5 的陷阱来源，ADR-039），
      逐个源陷阱在它周围的可放置区域上积分（`TrapCaptureGeneration`），与生成网格无关。

    `temperature_field` 是泊松网格上的温度场（热反馈时取 T14 的解，§6.6）：单元事件取它在单元中心
    的值，源陷阱事件取它在每个积分节点上的值；R_1e 的声子线型、电极费米分布、陷阱项的 MA 因子与
    P_bb 都按局域温度求值。缺省用内层的工作点温度。
    """
    device, solver, defects = inner.device, inner.solver, inner.defects
    voltage = inner.sim.voltage
    material = device.stack.single_material
    q = constants.elementary_charge
    rho = solver.deposit(defects.positions, q * (defects.charge_state - occupancy))
    phi = solver.solve(rho, top_boundary_potential(device, voltage))
    traps = TrapCaptureGeneration(inner, occupancy, tddb, phi, temperature_field)

    rates = np.zeros(len(cells))
    live = np.flatnonzero(admissible_cells(cells, defects.positions, device))
    if live.size == 0 or tddb.precursor_concentration == 0.0:
        return GenerationTable(rates, traps)
    centers = cells.centers[live]
    local = temperature_field is not None
    temperature = solver.interpolate(temperature_field, centers) if local else inner.sim.temperature
    field = np.linalg.norm(
        solver.field_vectors_at(phi, centers, np.zeros(live.size, dtype=int)), axis=0
    )
    conduction = device.band_reference_profile(centers[:, 2]) - q * solver.interpolate(phi, centers)

    # 电极→单元的路径与势垒只依赖单元所在的 z 层，按层算好再按单元取
    z_nodes = solver.z
    band = conduction_band_edge(device, z_nodes, solver.lateral_average(phi))
    iface_idx, delta_Ec = _interface_step_nodes(device, z_nodes)
    t = np.linspace(0.0, 1.0, _TRAP_PATH_POINTS)
    z_layer = cells.z_centers[:, None]
    by_layer = []
    for path, electrode, fermi in (
        (z_layer * t[None, :], device.cathode, 0.0),
        (z_layer + (device.Lz - z_layer) * t[None, :], device.anode, anode_fermi_level(voltage)),
    ):
        by_layer.append(
            CaptureElectrode(
                path=path,
                barrier=_band_on_path_steps(path, z_nodes, band, iface_idx, delta_Ec),
                m_di_path=device.stack.m_di_at(path),
                fermi_level=fermi,
                band_bottom=electrode_band_bottom(fermi, electrode.fermi_energy),
                mass=electrode.effective_mass,
            )
        )

    capture = np.empty(live.size)
    for start in range(0, live.size, _CELL_CHUNK):
        rows = slice(start, start + _CELL_CHUNK)
        layer = cells.layer[live[rows]]
        electrodes = [
            side._replace(path=side.path[layer], barrier=side.barrier[layer], m_di_path=side.m_di_path[layer])
            for side in by_layer
        ]
        capture[rows] = single_electron_capture_rate(
            conduction[rows],
            field[rows],
            electrodes,
            tddb.precursor_ionization_energy,
            tddb.precursor_relaxation_energy,
            material.phonon_energy,
            material.band_gap,
            material.m_di,
            temperature[rows] if local else temperature,
            _PRECURSOR_N_PHONON,
        )
    per_precursor = bond_break_generation_rate(
        capture, tddb.activation_energy, tddb.dipole_moment, material.kappa, field, temperature
    )
    rates[live] = tddb.precursor_concentration * cells.volume * per_precursor
    return GenerationTable(rates, traps)


def _place_in_cell(
    cells: GenerationCells,
    cell: int,
    positions: np.ndarray,
    standoff: float,
    device: DeviceConfig,
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    """在单元内均匀随机落点，与已有缺陷的距离 ≥ standoff（xy 取最小镜像）；放不下返回 None。"""
    for _ in range(_MAX_PLACEMENT_ATTEMPTS):
        candidate = cells.lower[cell] + cells.size * rng.random(3)
        if standoff <= 0.0 or positions.shape[0] == 0:
            return candidate
        delta = minimum_image_delta(positions - candidate[None, :], device.Lx, device.Ly)
        if np.min(np.linalg.norm(delta, axis=-1)) >= standoff:
            return candidate
    return None


def _draw_generation(
    rates: np.ndarray,
    full: np.ndarray,
    cells: GenerationCells,
    traps: Optional[TrapCaptureGeneration],
    positions: np.ndarray,
    standoff: float,
    device: DeviceConfig,
    rng: np.random.Generator,
) -> Optional[Tuple[float, np.ndarray]]:
    """抽下一次生成事件，返回 (Δt, 新空位坐标)；总生成率为 0 时返回 None。

    `rates` 与 `full` 是单元事件在前、源陷阱事件（`traps`）在后的整张事件表；`traps` 为 None
    时只有单元事件。选中单元时在单元内均匀落点，选中源陷阱时按它周围的被积函数落点。选中的事件
    放不下新空位时把它记为已满（原地改 `full`），按剩下的事件重抽 Δt 与事件：缺陷只增不减，
    可放置区域只缩不扩，满了的单元与源陷阱一直是满的。
    """
    while True:
        live = np.where(full, 0.0, rates)
        total = float(live.sum())
        if not total > 0.0:
            return None
        dt = -np.log(rng.random()) / total
        cumulative = np.cumsum(live)
        event = int(np.searchsorted(cumulative, rng.random() * cumulative[-1], side="right"))
        event = min(event, live.size - 1)
        if event < len(cells):
            position = _place_in_cell(cells, event, positions, standoff, device, rng)
        else:
            position = traps.sample(event - len(cells), rng)
        if position is not None:
            return dt, position
        full[event] = True


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


def percolation_clusters(
    positions: np.ndarray, device: DeviceConfig, distance: float
) -> Tuple[np.ndarray, np.ndarray]:
    """渗流团簇（§6.3）：间距 < `distance` 的缺陷连边后的连通分量。

    返回 (各缺陷所在分量的编号 (N,), 各分量是否同时触到两个电极 (分量数,))。电极当作一个
    节点、与缺陷按同一截断连边：缺陷离电极界面 < `distance` 就算触到该电极。xy 取最小镜像。
    与强耦合簇（按速率连边、用于电流估计，ADR-022）是两回事。
    """
    n = positions.shape[0]
    if n == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=bool)
    pairs = neighbor_pairs(positions, device.Lx, device.Ly, distance)
    graph = coo_matrix((np.ones(len(pairs)), (pairs.a, pairs.b)), shape=(n, n))
    count, labels = connected_components(graph, directed=False)
    z = positions[:, 2]
    cathode = np.bincount(labels, z < distance, count) > 0
    anode = np.bincount(labels, z > device.Lz - distance, count) > 0
    return labels, cathode & anode


@dataclass
class TDDBResult:
    """一条 TDDB 瞬态轨迹（恒压应力）。

    第 k 条记录是第 k 次生成之后的输运稳态（k = 0 为 t = 0 的初始缺陷集合），电流在
    [time[k], time[k+1]) 上保持不变，最后一条保持到 `end_time`。电流均为净值，正号 = 电子由
    阴极流向阳极。
    """

    voltage: float  # [V]
    temperature: float  # 环境温度（电极恒温）[K]
    area: float  # 电极面积 [m^2]
    seed: int
    time: np.ndarray  # (K,) 各记录的时刻 [s]
    current: np.ndarray  # (K,) [A]
    by_mechanism: Dict[str, np.ndarray]  # 各机制 (K,) [A]，四项之和 = current
    generation_rate: np.ndarray  # (K,) 这一状态下的总生成速率 ΣG [1/s]：单元事件 + 源陷阱事件（已满的不计）
    n_defects: np.ndarray  # (K,) 缺陷总数（初始 + 生成）
    space_charge: np.ndarray  # (K,) 全部缺陷的净电荷 q·Σ(Z − p) [C]（§8.6）
    empty_generated: np.ndarray  # (K,) 生成的空位里处于空态的期望个数 Σ(1 − p)（§8.6）
    iterations: np.ndarray  # (K,) 平均场求 F 的次数
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
    # --- 击穿判据（§6.3）---
    percolation_size: np.ndarray  # (K,) 最大渗流团簇（间距 < d_c）的缺陷数
    dense_cluster_size: np.ndarray  # (K,) 最大致密团簇（间距 < 6 Å，论文的标志）的缺陷数
    end_time: float  # 轨迹终点 [s]；击穿时就是 t_BD
    # "breakdown"（渗流）| "compliance"（限流）| "max_generated" | "stop_time" | "no_generation"
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
        """渗流判据触发的时刻 t_BD [s]；没有击穿时为 None。"""
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
        """任意时刻 t ∈ [0, end_time] 的电流 [A]（分段常数）。"""
        index = np.searchsorted(self.time, np.asarray(times, dtype=float), side="right") - 1
        return self.current[np.clip(index, 0, None)]


def run_tddb(
    config: Config, progress: Optional[Callable[[int, float, float], None]] = None
) -> TDDBResult:
    """按 `config` 跑一条 TDDB 瞬态轨迹（模块文档的双层循环）。

    初始缺陷与 `KMCSimulation(config)` 是同一个实现：同一个种子的随机数流先生成初始缺陷，
    之后的生成事件接着用同一个流。应力工况取 `simulation.voltage`，环境温度取
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
    solver = PoissonSolver(device, sim.grid)
    coupling = IncrementalCoupling.build(solver, defects.positions, defects.field_side)
    cells = generation_cells(device, tddb.generation_grid)
    standoff = _global_standoff(device)
    # 事件表 = 单元事件 + 每个缺陷一条源陷阱事件；放不下新空位的事件记为已满，缺陷只增不减，一直满着
    full = np.zeros(len(cells) + n_initial, dtype=bool)
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
        table = generation_rates(inner, occupancy, tddb, cells, temperature_field)
        rates = np.where(full, 0.0, np.concatenate([table.cells, table.traps.rates]))
        n_clusters, max_cluster, largest, n_split = _cluster_diagnostics(inner)
        labels, spanning = percolation_clusters(defects.positions, device, tddb.percolation_distance)
        dense_labels, _ = percolation_clusters(defects.positions, device, _DENSE_CLUSTER_DISTANCE)
        records.append(
            dict(
                time=time,
                current=mf.current,
                by_mechanism=mf.by_mechanism,
                generation_rate=float(rates.sum()),
                n_defects=len(defects),
                space_charge=constants.elementary_charge * float(np.sum(defects.charge_state - occupancy)),
                empty_generated=float(np.sum(1.0 - occupancy[n_initial:])),
                iterations=mf.iterations,
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
                percolation_size=int(np.bincount(labels).max(initial=0)),
                dense_cluster_size=int(np.bincount(dense_labels).max(initial=0)),
            )
        )
        if progress is not None:
            progress(len(defects) - n_initial, time, mf.current_density)

        if np.any(spanning):
            end_time, stop_reason = time, "breakdown"
            break
        if mf.current_density >= tddb.compliance_current_density:
            end_time, stop_reason = time, "compliance"
            break
        if len(defects) - n_initial >= tddb.max_generated_vacancies:
            end_time, stop_reason = time, "max_generated"
            break
        event = _draw_generation(rates, full, cells, table.traps, defects.positions, standoff, device, rng)
        if event is None:
            end_time, stop_reason = tddb.stop_time, "no_generation"
            break
        dt, position = event
        if time + dt > tddb.stop_time:
            end_time, stop_reason = tddb.stop_time, "stop_time"
            break
        time += dt
        defects = append_defects(defects, layer_bulk_defects(device, 0, position, rng))
        coupling = coupling.extended(position)
        full = np.append(full, False)
        occupancy = np.append(occupancy, 1.0)  # §6.7：出生即被一个电子占据，电荷态为 0
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
        by_mechanism={
            name: np.array([record["by_mechanism"][name] for record in records]) for name in MECHANISMS
        },
        generation_rate=column("generation_rate"),
        n_defects=column("n_defects", int),
        space_charge=column("space_charge"),
        empty_generated=column("empty_generated"),
        iterations=column("iterations", int),
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
        percolation_size=column("percolation_size", int),
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
    返回 (拟合结果, 取样时刻)；拟合的电流单位是 A/m²。
    """
    if t_start is None:
        if result.time.size < 2:
            raise ValueError("轨迹里没有生成事件，无法做 I-t 拟合")
        t_start = float(result.time[1])
    t_stop = result.end_time if t_stop is None else t_stop
    decades = np.log10(t_stop / t_start)
    times = np.logspace(np.log10(t_start), np.log10(t_stop), max(2, int(np.ceil(decades * points_per_decade)) + 1))
    return fit_power_law(times, result.current_at(times) / result.area), times


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
                "generation_rate_s", "space_charge_C", "empty_generated",
                "mf_iterations", "mf_residual", "mf_converged",
                "n_clusters", "max_cluster_size", "largest_component", "n_split",
                "power_W", "T_transport_K", "T_effective_K", "T_peak_K",
                "percolation_size", "dense_cluster_size",
            ]
        )
        for k in range(result.time.size):
            writer.writerow(
                ["{:.6e}".format(result.time[k]), "{:.6e}".format(result.segment_end[k])]
                + [str(int(result.n_generated[k])), str(int(result.n_defects[k])), "{:.6e}".format(density[k])]
                + ["{:.6e}".format(result.by_mechanism[name][k] / result.area * per_cm2) for name in MECHANISMS]
                + [
                    "{:.6e}".format(result.generation_rate[k]),
                    "{:.6e}".format(result.space_charge[k]),
                    "{:.6g}".format(result.empty_generated[k]),
                    str(int(result.iterations[k])),
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
                    str(int(result.percolation_size[k])),
                    str(int(result.dense_cluster_size[k])),
                ]
            )
