"""TDDB 瞬态驱动（等温）：准静态双层循环。

规格见 PROJECT_CONTEXT_TDDB.md §5、§6，决策记录见 ADR-032–034。公式编号 T1–T4 是 TDDB 的
编号（PROJECT_CONTEXT_TDDB.md 第 3 节），与 docs/rate_equations.md 的 R1–R8 不对应。

电子输运事件的时间尺度（~10⁻¹³…10⁻⁶ s）比缺陷生成（秒以上）短十几个数量级，所以输运相对
生成瞬间达到稳态（§5）。外层是缺陷生成的 Gillespie 循环，每步：

1. **内层稳态**：按当前缺陷集合重建 `KMCSimulation`（静电耦合用 `IncrementalCoupling` 逐个
   追加的那份），从上一步的占据热启动，求簇级平均场不动点（`mean_field.solve_mean_field`），
   得到电流、分解与各缺陷占据 p。近邻表、强耦合簇与截断半径都随缺陷集合从头重算，与对同一
   缺陷集合新建的 `KMCSimulation` 逐项相同（ADR-033）。
2. **生成率表**（§6.1 模型 C）：各缺陷净电荷 q·(Z − p) 与外加偏压一起解一次泊松方程，在每个
   生成单元中心对一个虚拟前驱求单步俘获速率 R_1e（式 T4，两侧电极 R2 + R4，与引擎处理真实
   陷阱同一口径）与断键概率 P_bb（式 T2），单元生成率 G_cell = N_T·V_cell·R_1e·P_bb（式 T1）。
   单元中心离已有缺陷不到 min_separation 的单元放不下新空位，生成率记 0。
3. **抽事件**：Δt = −ln(r)/ΣG，按 G_cell 选单元，在单元内连续随机落点，与已有缺陷的距离小于
   min_separation 就在单元内重抽（§6.1、§6.8）。
4. **出生**：新空位与初始缺陷同一物种（能级按缺陷预设的分布抽样，Z、S、m_di 取预设），出生
   占据 = 1（§6.7）：它此刻净电荷为 0，不改变其余缺陷处的电势与电场。静电耦合增量扩一行
   一列（§6.4），回到 1。

温度固定为环境值（Phase 3 等温，§6.6）；击穿判据（渗流）属于 Phase 4，这里按生成空位数上限或
应力时间上限停止。

**电荷记账**：新空位出生时带的那个电子来自前驱俘获，不经过电极，不计入输运电流（§6.7）。
平均场没有时间推进，每条记录的电流就是该缺陷集合下的稳态电流。
"""

import csv
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

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
    minimum_image_delta,
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
from .rates import CaptureElectrode, bond_break_generation_rate, single_electron_capture_rate

# 前驱俘获 R4 的声子项数：E_rel1 = 1.0 eV（E_rel1/ℏω₀ = 25）时 80 项最大相对误差 1e-4、
# 100 项到机器精度（ADR-030）
_PRECURSOR_N_PHONON = 100
# 生成单元分块求 R_1e：每块 (块长, 声子数, 路径点) 的数组约 0.4 MB，留在缓存里。16×16×70 单元上
# 实测单进程 256 块 4.3 s、8 块 3.6 s；16 个进程并发时 256 块 36 s（内存带宽占满）、8 块 9–14 s
_CELL_CHUNK = 8
# 选中单元里落点的尝试次数上限，全部离已有缺陷太近就把单元记为已满
_MAX_PLACEMENT_ATTEMPTS = 1000


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


def generation_rates(
    inner: KMCSimulation, occupancy: np.ndarray, tddb: TDDBConfig, cells: GenerationCells
) -> np.ndarray:
    """各生成单元的 O 空位生成速率 G_cell = N_T·V_cell·R_1e·P_bb [1/s]（式 T1，§6.1 模型 C）。

    `inner` 提供当前缺陷集合、工作点与泊松求解器，`occupancy` 是各缺陷的平均场占据 p。
    每个缺陷的净电荷 q·(Z − p) 与外加偏压一起解一次泊松方程（与引擎用耦合矩阵线性叠加的
    结果相同），在单元中心对一个虚拟前驱求值，取法同 `kmc.KMCSimulation._rates` 处理真实陷阱：

    - 前驱处导带底 E_C = B − e·φ 定能级 E = E_C − E_t1，局域场大小进 R4 的 c_MP 与 P_bb；
    - 电极→单元的 WKB 路径沿 z 直线，势垒取横向平均剖面的导带底；
    - R_1e = Σ_两侧电极 (R2 + R4)（式 T4，`single_electron_capture_rate`，R4 取
      `_PRECURSOR_N_PHONON` 个声子项）；P_bb 的场加速项 p₀·(2+k)/3·E 里 k 取静态 κ（式 T2）。

    单元中心落在已有缺陷 min_separation 之内的单元记 0（`admissible_cells`）：那里放不下新空位，
    而带电缺陷近旁的库仑场经 (2+κ)/3 放大后会把 P_bb 压到饱和，把这个速率算给整个单元会让
    ΣG 随生成网格加密发散（ADR-033）。
    """
    device, solver, defects = inner.device, inner.solver, inner.defects
    voltage, temperature = inner.sim.voltage, inner.sim.temperature
    material = device.stack.single_material
    q = constants.elementary_charge
    rates = np.zeros(len(cells))
    live = np.flatnonzero(admissible_cells(cells, defects.positions, device))
    if live.size == 0 or tddb.precursor_concentration == 0.0:
        return rates

    rho = solver.deposit(defects.positions, q * (defects.charge_state - occupancy))
    phi = solver.solve(rho, top_boundary_potential(device, voltage))
    centers = cells.centers[live]
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
            temperature,
            _PRECURSOR_N_PHONON,
        )
    per_precursor = bond_break_generation_rate(
        capture, tddb.activation_energy, tddb.dipole_moment, material.kappa, field, temperature
    )
    rates[live] = tddb.precursor_concentration * cells.volume * per_precursor
    return rates


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
    positions: np.ndarray,
    standoff: float,
    device: DeviceConfig,
    rng: np.random.Generator,
) -> Optional[Tuple[float, np.ndarray]]:
    """抽下一次生成事件，返回 (Δt, 新空位坐标)；总生成率为 0 时返回 None。

    选中的单元放不下新空位时把它记为已满（原地改 `full`），按剩下的单元重抽 Δt 与单元：
    缺陷只增不减，满了的单元一直是满的。
    """
    while True:
        live = np.where(full, 0.0, rates)
        total = float(live.sum())
        if not total > 0.0:
            return None
        dt = -np.log(rng.random()) / total
        cumulative = np.cumsum(live)
        cell = int(np.searchsorted(cumulative, rng.random() * cumulative[-1], side="right"))
        cell = min(cell, live.size - 1)
        position = _place_in_cell(cells, cell, positions, standoff, device, rng)
        if position is not None:
            return dt, position
        full[cell] = True


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


@dataclass
class TDDBResult:
    """一条 TDDB 瞬态轨迹（恒压应力、等温）。

    第 k 条记录是第 k 次生成之后的输运稳态（k = 0 为 t = 0 的初始缺陷集合），电流在
    [time[k], time[k+1]) 上保持不变，最后一条保持到 `end_time`。电流均为净值，正号 = 电子由
    阴极流向阳极。
    """

    voltage: float  # [V]
    temperature: float  # [K]
    area: float  # 电极面积 [m^2]
    seed: int
    time: np.ndarray  # (K,) 各记录的时刻 [s]
    current: np.ndarray  # (K,) [A]
    by_mechanism: Dict[str, np.ndarray]  # 各机制 (K,) [A]，四项之和 = current
    generation_rate: np.ndarray  # (K,) 这一状态下的总生成速率 ΣG [1/s]（已满单元不计）
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
    end_time: float  # 轨迹终点 [s]
    stop_reason: str  # "max_generated" | "stop_time" | "no_generation"
    n_initial: int  # 初始缺陷个数：编号 < n_initial 的是初始缺陷，其余是应力生成的空位
    birth_time: np.ndarray  # (N,) 各缺陷的出生时刻 [s]，初始缺陷记 0
    defects: DefectEnsemble  # 终态缺陷集合
    occupancy: np.ndarray  # 终态的平均场占据

    @property
    def current_density(self) -> np.ndarray:
        """各记录的电流密度 [A/m^2]。"""
        return self.current / self.area

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
    之后的生成事件接着用同一个流。应力工况取 `simulation.voltage` / `temperature`，生成参数取
    `config.tddb`。本阶段只做单层（PROJECT_CONTEXT_TDDB.md §1）。

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
    full = np.zeros(len(cells), dtype=bool)
    birth_time = np.zeros(n_initial)
    occupancy: Optional[np.ndarray] = None
    time = 0.0
    records: List[dict] = []

    while True:
        inner = KMCSimulation(config, defects=defects, coupling=coupling, solver=solver)
        mf = solve_mean_field(inner, occupancy)
        occupancy = mf.occupancy
        rates = np.where(full, 0.0, generation_rates(inner, occupancy, tddb, cells))
        n_clusters, max_cluster, largest, n_split = _cluster_diagnostics(inner)
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
            )
        )
        if progress is not None:
            progress(len(defects) - n_initial, time, mf.current_density)

        if len(defects) - n_initial >= tddb.max_generated_vacancies:
            end_time, stop_reason = time, "max_generated"
            break
        event = _draw_generation(rates, full, cells, defects.positions, standoff, device, rng)
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
        occupancy = np.append(occupancy, 1.0)  # §6.7：出生即占据，此刻净电荷为 0
        birth_time = np.append(birth_time, time)

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
        end_time=end_time,
        stop_reason=stop_reason,
        n_initial=n_initial,
        birth_time=birth_time,
        defects=defects,
        occupancy=occupancy,
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
                ]
            )
